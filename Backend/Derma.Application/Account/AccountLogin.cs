using Derma.Application.Core;
using Derma.Domain;
using FluentValidation;
using MediatR;
using Microsoft.AspNetCore.Identity;

namespace Derma.Application.Account
{
    public class AccountLogin
    {
        public class Command : IRequest<string?>
        {
            public required string Email { get; set; }
            public required string Password { get; set; }
        }

        public class CommandValidator : AbstractValidator<Command>
        {
            public CommandValidator()
            {
                RuleFor(x => x.Email).NotEmpty().WithMessage("Email jest wymagany.");
                RuleFor(x => x.Password).NotEmpty().WithMessage("Hasło jest wymagane.");
            }
        }

        public class Handler : IRequestHandler<Command, string?>
        {

            private readonly UserManager<AppUser> _userManager;
            private readonly ITokenService _tokenService;

            public Handler(UserManager<AppUser> userManager, ITokenService tokenService)
            {

                _userManager = userManager;
                _tokenService = tokenService;

            }

            public async Task<string?> Handle(Command request, CancellationToken cancellationToken)
            {
                var user = await _userManager.FindByEmailAsync(request.Email);

                if (user == null || !await _userManager.CheckPasswordAsync(user, request.Password))
                    return null;
                var roles = await _userManager.GetRolesAsync(user);
                return _tokenService.CreateToken(user, roles);

            }
        }
    }
}